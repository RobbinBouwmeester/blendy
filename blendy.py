# sklearn imports
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import cross_val_score

from sklearn.metrics import *

# numpy imports
from numpy.random import randint

# pandas imports
import pandas

# python imports
import warnings
from operator import itemgetter
from timer import Timer

class Blender():
	"""
	
	Class that is a wrapper for sklearn to enable blending different models.

	!!! Currently only suitable for classification !!!

	"""

	def __init__(self,X,y,nfolds=10,fold_list=None,identifiers=None,
				 verbose=True,classification=True,max_train=None):
		"""

		Initialize class with the folowing parameters:

			X 			- pandas.DataFrame that contains the features (/variables) as columns and rows as instances
			y 			- objective value as pandas.Series objext
			nfolds 		- number of folds to divide the instances in
			fold_list 	- instead of randomly assigning folds, specificy your own folds in a list, make sure the 
						  indexes are in the same order as the instances in X and the supplied identifiers.
			identifiers - supply your own list of identifiers, making sure that they are ordered the same as
						  X and (if specified) the fold_list. Make sure identifiers are unique.
			verbose 	- verbosity as boolean
			max_train	- maximum of training instances to consider, remaining will be used for testing

		"""
		
		self.X = X
		self.y = y
		self.nfolds = nfolds

		# TODO make suitable for regression tasks
		# If classification check the possible outcome classes
		self.classification = classification
		if classification: self.classes = list(set(self.y))
		
		self.max_train = max_train

		
		# If identifiers are specified, check if identifiers are unique. Otherwise
		# make own identifiers that are simply "ID" with a number that specifiers the row.
		if identifiers:
			if len(identifiers) != len(set(identifiers)): raise StandardError("Please supply unique identifiers.")
			self.identifiers = pandas.Series(identifiers)
		else: self.identifiers = pandas.Series(["ID%s" % (i) for i in range(len(y))])

		self.X.index = self.identifiers
		self.y.index = self.identifiers

		# Set specified fold or get folds
		if fold_list: self.fold_list = pandas.Series(fold_list)
		else: self.fold_list = pandas.Series(self._makefolds(max_train=self.max_train,nfolds=self.nfolds))

		# Make a dictionary that translates identifiers of instances to folds
		self.id_to_fold = dict(zip(self.identifiers,self.fold_list))

		# Make a dictionary that translates folds to identifiers
		self.fold_to_id = self._get_fold_dict()

		# Set universal counters for models
		self.universal_model_counter = 1
		self.universal_model_counter_blend = 1

		# Instantiate variables that are related to models and their computation
		self.models_params = {}
		self.blend_models_params = {}
		self.trained_models = {}
		self.trained_models_blend = {}
		self.trainpreds = {}

		self.preds_df = pandas.DataFrame()

		self.verbose = verbose

		if self.verbose: 
			t = Timer()
			self.t = t
			print self.t.start(message="Object initialized")

	def __str__(self):
		return("\n\nHi, I'm blendy. I was created on:\n\n%s\n\n" % (self.t.elapsed(message="")))

	def _makefolds(self,max_train=None,nfolds=10):
		"""

		Internal function that should not be called... Divides the instances in folds.

		With the following parameters:

			max_train	- integer that indicates the maximum number of instances to train on
			nfolds 		- number of folds that should be used

		Returns:

			list of integers that specify the fold number
		
		"""

		# If there is no maximum just divide into folds
		if max_train == None: return(randint(0,self.nfolds,len(self.y)))

		# If max _train is specified calculate how many hidden folds we need to make.
		# Further down the lins the fold numbers that are higher than the specified folds are ignored.
		per_fold = max_train/float(nfolds)
		needed_folds = len(self.y)/per_fold
		return(randint(0,needed_folds,len(self.y)))

	def _get_fold_dict(self):
		"""

		Internal function that should not be called... Returns a dictionary that maps the identifiers to folds.

		With the following parameters:

			-

		returns:

			dictionary that maps folds (key) to instances (value)
		
		"""

		fold_dict = {}
		for i in range(len(self.identifiers)):
			if fold_dict.has_key(self.fold_list[i]): fold_dict[self.fold_list[i]].append(self.identifiers[i])
			else: fold_dict[self.fold_list[i]] = [self.identifiers[i]]
		return(fold_dict)


	def add_model(self,model,params,blend=False,name=None):
		"""

		Adding a model in the first layer of blendy for training.

		With the following parameters:

			model 	- a sklearn object that resembles the machine learning algorithm
			params 	- dictionary with the parameters for the earlier defined object
			blend 	- boolean that determines if the model is used for blending in the last step
					  to go from multiple predictions to a single prediction.
			name 	- define a name for the model

		returns:

			-
		
		"""

		if name: 
			if name in self.models_params.keys() or name in self.blend_models_params.keys(): 
				name = "%s_%s" % (name,self.universal_model_counter)
				self.universal_model_counter += 1
				warnstr = "Name for model already exists. Will use counter for new name: %s" % (name)
				warnings.warn(warnstr, UserWarning)
			if blend: self.blend_models_params[name] = [model,params]
			else: self.models_params[name] = [model,params]

		else:
			if blend: self.blend_models_params["blendmodel_num_%s" % self.universal_model_counter_blend] = [model,params]
			else: self.models_params["blendmodel_num_%s" % self.universal_model_counter] = [model,params]
			self.universal_model_counter += 1

	# TODO make it possible to use a different scoring when calculating the performance
	def pretrain(self,retrain=True,train_preds=True,calc_perf=True,scorer=log_loss):
		"""

		Adding a model in the first layer of blendy for training.

		With the following parameters:

			retrain		- ignore earlier training and overwrite
			train_preds	- write results so they can be used in the next layer?
			calc_perf	- calculate the performance
			scores 		- name of scoring function, make sure this complies with naming of sklearn

		returns:

			-
		
		"""

		if len(self.models_params.keys()) == 0:
			raise StandardError("Please specify models and their rescpective parameters first.")

		for k in self.models_params.keys():
			if k in self.trained_models and retrain: continue
			if train_preds and retrain: self.trainpreds[k] = {}
			if calc_perf: mean_perf = 0

			if self.verbose: print self.t.elapsed(message="Training model %s" % (k))

			for i in range(self.nfolds):
				train_folds = range(self.nfolds)
				train_folds.remove(i)

				train_X = self._get_x(include_folds=train_folds)
				valid_X = self._get_x(exclude_folds=train_folds)
				train_y = self._get_y(include_folds=train_folds)
				valid_y = self._get_y(exclude_folds=train_folds)

				# Making sure the sorting is the same for instances and objective values
				train_y = train_y[train_X.index]
				valid_y = valid_y[valid_X.index]

				model = self.train_model(k,X=train_X,y=train_y)
				if train_preds and self.classification:
					class_preds = self._preds_train_fold(model,valid_X)
					temp_classes = self._get_identifiers_folds(exclude_folds=train_folds)

					# TODO maybe do something with a cleaner lambda and map below?
					for j in range(len(class_preds)):
						self.trainpreds[k][temp_classes[j]] = class_preds[j]
				if calc_perf:
					lloss = scorer(valid_y,model.predict_proba(valid_X))
					mean_perf += lloss 
					print self.t.elapsed(message="Performance on fold(s) %s for model %s: %s" % (i,k,lloss))
			if calc_perf:
				print self.t.elapsed(message="Mean performance for model %s: %s" % (k,mean_perf/float(self.nfolds)))
			model = self.train_model(k,X=self.X,y=self.y)
			self.trained_models[k] = model

	def _preds_train_fold(self,model,X):
		"""

		Internal function that should not be called... Gets predictions from a specified model and makes sure
		that the order of the returned pandas.DataFrame comply with the classes.

		With the following parameters:

			model - the model that is used for predicting
			X 	  - the instances used for predicting

		returns:

			two-dimensional list with predictions per class
		
		"""

		preds_train = model.predict_proba(X)
		if not list(model.classes_) == self.classes:
			new_order = [self.classes.index(c) for c in list(model.classes_)]
			preds_train = map(itemgetter(*new_order),preds_train)
		return(preds_train)

	def _get_x(self,include_folds=[],exclude_folds=[]):
		"""

		Internal function that should not be called... Returns instances from specified folds.

		With the following parameters:

			include_folds - list with folds to include
			exclude_folds - list with folds to exclude

		returns:

			pandas.DataFrame object with specified instances
		
		"""

		if len(include_folds) == 0 and len(exclude_folds) == 0:
			raise StandardError("Please specify folds to exclude or include.")
		if len(include_folds) != 0 and len(exclude_folds) != 0:
			raise StandardError("Please specify include or exclude folds.")
		
		idents = self._get_identifiers_folds(include_folds=include_folds,exclude_folds=exclude_folds)
		return(self.X.loc[idents,])

	def _get_y(self,include_folds=[],exclude_folds=[]):
		"""

		Internal function that should not be called... Returns objective values from specified folds.

		With the following parameters:

			include_folds - list with folds to include
			exclude_folds - list with folds to exclude

		returns:

			pandas.Series object with specified objective values
		
		"""

		if len(include_folds) == 0 and len(exclude_folds) == 0:
			raise StandardError("Please specify folds to exclude or include.")
		if len(include_folds) != 0 and len(exclude_folds) != 0:
			raise StandardError("Please specify include or exclude folds.")
		
		idents = self._get_identifiers_folds(include_folds=include_folds,exclude_folds=exclude_folds)
		return(self.y.loc[idents])

	def _get_identifiers_folds(self,include_folds=[],exclude_folds=[]):
		"""

		Internal function that should not be called... Returns identifiers for particular folds in a list.

		With the following parameters:

			include_folds - list with folds to include
			exclude_folds - list with folds to exclude

		returns:

			list with identifiers for specified folds
		
		"""

		if len(include_folds) == 0 and len(exclude_folds) == 0:
			raise StandardError("Please specify folds to exclude or include.")
		if len(include_folds) != 0 and len(exclude_folds) != 0:
			raise StandardError("Please specify include or exclude folds.")
		
		idents = []
		if len(include_folds) > 0:
			for i in include_folds:
				idents.extend(self.fold_to_id[i])
		else:
			include_folds = list(set(self.fold_to_id.keys()) - set(exclude_folds))
			for e in include_folds:
				idents.extend(self.fold_to_id[e])
		return(idents)


	def train_model(self,name,X,y):
		"""

		Train a model for specified instances and objective values.

		With the following parameters:

			name 	- name of the added model
			X 		- list with folds to include
			y 		- objective values

		returns:

			sklearn object that contains the trained model
		
		"""

		if name not in self.models_params.keys(): raise StandardError("Unable to retrieve the model: %s" % (name))
		model,params = self.models_params[name]
		model.set_params(**params)
		model.fit(X,y)
		return(model)

	def train_blend(self,scorer="log_loss"):
		"""

		Train a blending model that blends predictions.

		With the following parameters:

			scorer	- scoring function used in cross-validation

		returns:

			-
		
		"""

		if len(self.models_params.keys()) == 0:
			raise StandardError("Please specify models and their rescpective parameters first.")

		for k in self.blend_models_params.keys():
			if self.verbose: print self.t.elapsed(message="Starting ")
			model,params = self.blend_models_params[k]
			
			X = self.preds_df
			y = list(map(self.get_y,list(self.preds_df.index)))
			

			cv = StratifiedKFold(y,n_folds=10)#,shuffle=True) #,random_state=i,shuffle=True)#,shuffle=True
			grid = GridSearchCV(model, params,cv=cv,scoring=scorer,verbose=0,n_jobs=1) #
			grid.fit(X,y)

			if self.verbose:
				for l in grid.grid_scores_:
					print self.t.elapsed(message=l)

			self.trained_models_blend[k] = grid

	def combine_train_preds(self,return_existing=False):
		"""

		Combine predictions from the models and return a pandas.DataFrame.

		With the following parameters:

			return_existing	- boolean indicating if it should return the existing dataframe

		returns:

			pandas.DataFrame object with all predictions
		
		"""

		if len(self.trainpreds.keys()) == 0: 
			raise StandardError("Predictions are not available. Probably because no predictions were made.")
		if return_existing: return(self.preds_df)

		for k in self.trainpreds.keys():
			temp_frame = pandas.DataFrame()
			temp_ident = []
			for ident_index in range(len(self.identifiers)):
				if self.fold_list[ident_index] not in range(self.nfolds): continue
				temp_ident.append(self.identifiers[ident_index])
				temp_frame = pandas.concat([temp_frame,pandas.DataFrame(self.trainpreds[k][self.identifiers[ident_index]])],axis=1,join="inner")	
			temp_frame = temp_frame.transpose()
			temp_frame.index = temp_ident
			temp_frame.columns = ["%s_%s" % (c,k) for c in self.classes]
			self.preds_df = pandas.concat([self.preds_df,temp_frame],axis=1,join="inner")
		
		return(self.preds_df)

	def set_identifiers(self,identifiers):
		"""

		Set new identifiers.

		With the following parameters:

			identifiers	- list of identifiers (in the same order as instances!)

		returns:

			-
		
		"""

		if len(identifiers) != len(set(identifiers)): raise StandardError("Please supply unique identifiers.")
		if len(identifiers) != len(self.X.index):  raise StandardError("Please supply identifiers that have the same length as the number of rows.")
		self.identifiers = pandas.Series(identifiers)

		self.X.index = self.identifiers
		self.y.index = self.identifiers

		self.id_to_fold = dict(zip(self.identifiers,self.fold_list))
		self.fold_to_id = self._get_fold_dict()


	def get_identifiers(self):
		"""

		Get existing identifiers.

		With the following parameters:

			-

		returns:

			a pandas.Series object with identifiers
		
		"""

		return(self.identifiers)

		
	def set_params(self,name,params):
		"""

		Set parameters for a model.

		With the following parameters:

			name 	- name of model
			params 	- new parameters for model

		returns:

			-
		
		"""

		try: blend_models_params[name][1] = params
		except KeyError:
			warnstr = "Could not set the params for model: %s" % (name)
			warnings.warn(warnstr, UserWarning)

	def get_params(self,name):
		"""

		Get parameters for a model.

		With the following parameters:

			name 	- name of model

		returns:

			dictionary with parameters
		
		"""

		try: return(blend_models_params[name][1])
		except KeyError:
			warnstr = "Could not get the params for model: %s" % (name)
			warnings.warn(warnstr, UserWarning)

	def get_y(self,name):
		"""

		Get y for a given identifier.

		With the following parameters:

			name 	- identifier of objective value

		returns:

			the y-values
		
		"""

		return(self.y[list(self.identifiers).index(name)])

	def get_trained_model(self,name):
		"""

		Get a trained model for a given identifier.

		With the following parameters:

			name 	- identifier of objective value

		returns:

			return a trained sklearn object
		
		"""

		try: return(self.trained_models[name])
		except KeyError: raise KeyError("The specified model could not be retrieved by name, it does not exist and/or is not trained yet.") 

	def get_trained_models(self):
		"""

		Get all trained models.

		With the following parameters:

			-

		returns:

			return all trained sklearn object
		
		"""

		return(self.trained_models)

	def get_preds_pretrain(self,X):
		"""

		Get all predictions for the pre-train.

		With the following parameters:

			-

		returns:

			return a pandas.DataFrame object with all predictions
		
		"""

		preds_df = pandas.DataFrame()
		for k in self.trained_models.keys():
			temp_frame = pandas.DataFrame(self.trained_models[k].predict_proba(X))
			temp_frame.columns = ["%s_%s" % (c,k) for c in self.classes]
			preds_df = pandas.concat([preds_df,temp_frame],axis=1,join="inner")
		return(preds_df)

	def get_preds_blend(self,X):
		"""

		Make predictions by blending existing predictions.

		With the following parameters:

			X - pandas.DataFrame with predictions that should be blended

		returns:

			return a pandas.DataFrame object with all predictions
		
		"""

		preds_df = pandas.DataFrame()
		for k in self.trained_models_blend.keys():
			temp_frame = pandas.DataFrame(self.trained_models_blend[k].predict_proba(X))
			temp_frame.columns = self.classes
			preds_df = pandas.concat([preds_df,temp_frame],axis=1,join="inner")
		return(preds_df)

	# TODO see below...

	#def get_models(self,):

	#def test_preds(self,):

	#def set_nfolds(self,):

	#def get_nfolds(self,):

	#def set_folds(self,):

	#def get_folds(self,):



if __name__ == "__main__":
	from sklearn import datasets
	from sklearn.ensemble import RandomForestClassifier
	from sklearn.linear_model import SGDClassifier

	iris = datasets.load_iris()
	X = pandas.DataFrame(iris.data[:, :3])  # we only take the first two features.
	y = pandas.Series(iris.target)

	#b1 = Blender(X,y,max_train=50)
	b1 = Blender(X,y)

	print b1

	b1.add_model(RandomForestClassifier(),
				{"n_estimators":1000,
				"min_samples_leaf":4},
				name="rf_nest1000")
	b1.add_model(RandomForestClassifier(),
				{"n_estimators":50,
				"min_samples_leaf":4},
				name="rf_nest50")

	b1.pretrain()

	b1.combine_train_preds()

	b1.add_model(SGDClassifier(loss="modified_huber"),
			{"loss":["modified_huber"],
			"penalty":["elasticnet"],
			"alpha":[1.0,1e-1,1e-2,1e-3,1e-4,1e-5,1e-6,1e-7,1e-8,1e-9,1e-10],
			"n_iter":[2,4,8,16,32,64,128]},
			name="rf_nest100_blend",
			blend = True)
	
	b1.train_blend()



	X2 = b1.get_preds_pretrain(X)
	print b1.get_preds_blend(X2)